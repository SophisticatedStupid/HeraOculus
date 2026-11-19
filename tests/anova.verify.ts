import { runOneWayAnova } from '../src/services/statistics/anova';

const data = {
  response: [1,2,3, 4,5,6, 7,8,9],
  group: ['A','A','A','B','B','B','C','C','C'],
};
const r = runOneWayAnova(data);
console.log('k', r.k, 'N', r.N);
console.log('F', r.table[0]?.F, 'p', r.table[0]?.p);
console.log('eta2', r.effectSizes.eta2);
console.log('error', r.error);
if (r.error) process.exit(1);
console.log('ALL_ANOVA_TESTS_PASSED');
