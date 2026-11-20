import { ptukey, qtukey, runOneWayAnova } from './src/services/statistics/anova';

console.log('Test 1: Two groups k=2');
const r1 = runOneWayAnova({ response: [1,2,3,4,5,6], group: ['A','A','A','B','B','B'] });
console.log('posthoc:', JSON.stringify(r1.posthoc));
console.log('equalVar:', r1.assumptions.equalVariancesAssumed);

console.log('\nTest 2: Three groups equal n');
const r2 = runOneWayAnova({ response: [1,2,3, 4,5,6, 7,8,9], group: ['A','A','A','B','B','B','C','C','C'] });
console.log('posthoc:', JSON.stringify(r2.posthoc, null, 2));

console.log('\nTest 3: ptukey/qtukey roundtrip');
for (const k of [2,3,4,5]) for (const df of [5,10,30,100]) {
  const p = 0.5;
  const q = qtukey(p, k, df);
  const p2 = ptukey(q, k, df);
  console.log('k='+k, 'df='+df, 'p='+p, 'q='+q.toFixed(4), 'p2='+p2.toFixed(4), 'err='+(Math.abs(p2-p).toFixed(6)));
}

console.log('\nTest 4: Unequal variances (should use Games-Howell)');
const r3 = runOneWayAnova({ 
  response: [1,1,1, 2,2,10, 3,3,3], 
  group: ['A','A','A','B','B','B','C','C','C'] 
});
console.log('posthoc:', JSON.stringify(r3.posthoc, null, 2));
console.log('equalVar:', r3.assumptions.equalVariancesAssumed);

console.log('\nTest 5: Small df');
const r4 = runOneWayAnova({ response: [1,2, 3,4], group: ['A','A','B','B'] });
console.log('posthoc:', JSON.stringify(r4.posthoc, null, 2));